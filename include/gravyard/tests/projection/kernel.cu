#include <cuda_runtime.h>
#include <iostream>

#include "camera/model.h"
#include "camera/opencv_pinhole.h"
#include "core/types.h"
#include "gaussian/primitive.h"
#include "projection/kernel.cuh"
#include "projection/operators/3dgs.h"

using namespace curend;

template <class T> T *create_device_ptr(const T &h_val) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T));
    cudaMemcpy(d_ptr, &h_val, sizeof(T), cudaMemcpyHostToDevice);
    return d_ptr;
}

template <class T> T *create_device_ptr() {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T));
    return d_ptr;
}

template <class T> void print_device_ptr(const T *d_ptr, const std::string &name) {
    T h_val;
    cudaMemcpy(&h_val, d_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << name << ": " << h_val << std::endl;
}

template <>
void print_device_ptr<glm::fvec2>(const glm::fvec2 *d_ptr, const std::string &name) {
    glm::fvec2 h_val;
    cudaMemcpy(&h_val, d_ptr, sizeof(glm::fvec2), cudaMemcpyDeviceToHost);
    std::cout << name << ": " << h_val.x << ", " << h_val.y << std::endl;
}

template <>
void print_device_ptr<glm::fvec3>(const glm::fvec3 *d_ptr, const std::string &name) {
    glm::fvec3 h_val;
    cudaMemcpy(&h_val, d_ptr, sizeof(glm::fvec3), cudaMemcpyDeviceToHost);
    std::cout << name << ": " << h_val.x << ", " << h_val.y << ", " << h_val.z
              << std::endl;
}

int main() {
    // setup camera
    auto const d_focal_length = create_device_ptr(glm::fvec2(800.0f, 600.0f));
    auto const d_principal_point = create_device_ptr(glm::fvec2(400.0f, 300.0f));
    auto const d_pose_r_start = create_device_ptr(glm::fmat3(1.f));
    auto const d_pose_t_start = create_device_ptr(glm::fvec3(0.f));
    std::array<uint32_t, 2> resolution = {800, 600};
    auto projector = OpencvPinholeProjection(d_focal_length, d_principal_point);
    auto d_camera = CameraModel(resolution, projector, d_pose_r_start, d_pose_t_start);

    // setup input gaussian
    auto const d_opacity = create_device_ptr(float(0.8f));
    auto const d_mean = create_device_ptr(glm::fvec3(0.0f, 0.0f, 1.0f));
    auto const d_quat = create_device_ptr(glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f));
    auto const d_scale = create_device_ptr(glm::fvec3(1.0f, 1.0f, 1.0f));
    auto const d_gaussian = DevicePrimitiveIn3DGS{d_mean, d_quat, d_scale, d_opacity};

    // setup output gaussian
    auto const d_opacity_out = create_device_ptr<float>();
    auto const d_mean_out = create_device_ptr<glm::fvec2>();
    auto const d_conic_out = create_device_ptr<glm::fvec3>();
    auto const d_depth_out = create_device_ptr<float>();
    auto const d_radius_out = create_device_ptr<glm::fvec2>();
    auto d_gaussian_out = DevicePrimitiveOut3DGS{
        d_opacity_out, d_mean_out, d_conic_out, d_depth_out, d_radius_out
    };

    // setup operator and launch kernel
    PreprocessOperator3DGS op;
    curend::device::PreprocessFwdKernel<false, 0><<<1, 256>>>(
        1,
        d_camera,
        1,
        d_gaussian,
        d_gaussian_out,
        op,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
    cudaDeviceSynchronize();

    // print output
    print_device_ptr(d_opacity_out, "Opacity");
    print_device_ptr(d_mean_out, "Mean");
    print_device_ptr(d_conic_out, "Conic");
    print_device_ptr(d_depth_out, "Depth");
    print_device_ptr(d_radius_out, "Radius");

    // free memory
    cudaFree(d_mean);
    cudaFree(d_quat);
    cudaFree(d_scale);
    cudaFree(d_opacity);

    cudaFree(d_opacity_out);
    cudaFree(d_mean_out);
    cudaFree(d_conic_out);
    cudaFree(d_depth_out);
    cudaFree(d_radius_out);

    cudaFree(d_focal_length);
    cudaFree(d_principal_point);
    cudaFree(d_pose_r_start);
    cudaFree(d_pose_t_start);

    return 0;
}