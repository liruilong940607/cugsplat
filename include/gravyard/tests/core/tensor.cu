#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "core/tensor.h"

using namespace cugsplat;

// Test kernel for MaybeCached with glm::fvec3
__global__ void test_tensor_vec3(MaybeCached<glm::fvec3, true> tensor) {
    // Each thread sets its own gradient
    glm::fvec3 grad(1.0f, 2.0f, 3.0f);
    tensor.set(grad);

    // Export gradient with warp reduction
    tensor.export_grad<32>();
}

// Test kernel for MaybeCached with glm::fmat3
__global__ void test_tensor_mat3(MaybeCached<glm::fmat3, true> tensor) {
    // Each thread sets its own gradient
    glm::fmat3 grad(1.0f);
    tensor.set(grad);

    // Export gradient with warp reduction
    tensor.export_grad<32>();
}

// Test kernel for MaybeCached with std::array<float, 6>
__global__ void test_tensor_float6(MaybeCached<std::array<float, 6>, true> tensor) {
    // Each thread sets its own gradient
    std::array<float, 6> grad = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensor.set(grad);

    // Export gradient with warp reduction
    tensor.export_grad<32>();
}

int main() {
    // Test with glm::fvec3
    {
        const int num_warps = 4;
        const int threads_per_warp = 32;
        const int total_threads = num_warps * threads_per_warp;

        glm::fvec3 *d_grad;
        cudaMalloc(&d_grad, sizeof(glm::fvec3));
        cudaMemset(d_grad, 0, sizeof(glm::fvec3));

        MaybeCached<glm::fvec3, true> tensor(d_grad);
        test_tensor_vec3<<<1, total_threads>>>(tensor);
        cudaDeviceSynchronize();

        glm::fvec3 h_grad;
        cudaMemcpy(&h_grad, d_grad, sizeof(glm::fvec3), cudaMemcpyDeviceToHost);

        std::cout << "fvec3 test:\n";
        std::cout << "Expected: (" << total_threads * 1.0f << ", "
                  << total_threads * 2.0f << ", " << total_threads * 3.0f << ")\n";
        std::cout << "Got: (" << h_grad.x << ", " << h_grad.y << ", " << h_grad.z
                  << ")\n";

        cudaFree(d_grad);
    }

    // Test with glm::fmat3
    {
        const int num_warps = 4;
        const int threads_per_warp = 32;
        const int total_threads = num_warps * threads_per_warp;

        glm::fmat3 *d_grad;
        cudaMalloc(&d_grad, sizeof(glm::fmat3));
        cudaMemset(d_grad, 0, sizeof(glm::fmat3));

        MaybeCached<glm::fmat3, true> tensor(d_grad);
        test_tensor_mat3<<<1, total_threads>>>(tensor);
        cudaDeviceSynchronize();

        glm::fmat3 h_grad;
        cudaMemcpy(&h_grad, d_grad, sizeof(glm::fmat3), cudaMemcpyDeviceToHost);

        std::cout << "\nfmat3 test:\n";
        std::cout << "Expected: all elements = " << total_threads * 1.0f << "\n";
        std::cout << "Got:\n";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_grad[i][j] << " ";
            }
            std::cout << "\n";
        }

        cudaFree(d_grad);
    }

    // Test with float[6]
    {
        const int num_warps = 4;
        const int threads_per_warp = 32;
        const int total_threads = num_warps * threads_per_warp;

        std::array<float, 6> *d_grad;
        cudaMalloc(&d_grad, sizeof(std::array<float, 6>));
        cudaMemset(d_grad, 0, sizeof(std::array<float, 6>));

        MaybeCached<std::array<float, 6>, true> tensor(d_grad);
        test_tensor_float6<<<1, total_threads>>>(tensor);
        cudaDeviceSynchronize();

        std::array<float, 6> h_grad;
        cudaMemcpy(
            &h_grad, d_grad, sizeof(std::array<float, 6>), cudaMemcpyDeviceToHost
        );

        std::cout << "\nstd::array<float, 6> test:\n";
        std::cout << "Expected: (" << total_threads * 1.0f << ", "
                  << total_threads * 2.0f << ", " << total_threads * 3.0f << ", "
                  << total_threads * 4.0f << ", " << total_threads * 5.0f << ", "
                  << total_threads * 6.0f << ")\n";
        std::cout << "Got: (";
        for (int i = 0; i < 6; ++i) {
            std::cout << h_grad[i];
            if (i < 5)
                std::cout << ", ";
        }
        std::cout << ")\n";

        cudaFree(d_grad);
    }

    return 0;
}