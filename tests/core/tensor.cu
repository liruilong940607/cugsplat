#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "core/tensor.h"

using namespace gsplat;

// Test kernel for Tensor with glm::fvec3
__global__ void test_tensor_vec3(Tensor<glm::fvec3> tensor) {
    // Each thread sets its own gradient
    glm::fvec3 grad(1.0f, 2.0f, 3.0f);
    tensor.set_grad(grad);

    // Export gradient with warp reduction
    tensor.export_grad<32>();
}

// Test kernel for Tensor with glm::fmat3
__global__ void test_tensor_mat3(Tensor<glm::fmat3> tensor) {
    // Each thread sets its own gradient
    glm::fmat3 grad(1.0f);
    tensor.set_grad(grad);

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

        Tensor<glm::fvec3> tensor(nullptr, d_grad);
        test_tensor_vec3<<<1, total_threads>>>(tensor);
        cudaDeviceSynchronize();

        glm::fvec3 h_grad;
        cudaMemcpy(&h_grad, d_grad, sizeof(glm::fvec3), cudaMemcpyDeviceToHost);

        std::cout << "fvec3 test:\n";
        std::cout << "Expected: (" << total_threads * 1.0f << ", "
                  << total_threads * 2.0f << ", " << total_threads * 3.0f
                  << ")\n";
        std::cout << "Got: (" << h_grad.x << ", " << h_grad.y << ", "
                  << h_grad.z << ")\n";

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

        Tensor<glm::fmat3> tensor(nullptr, d_grad);
        test_tensor_mat3<<<1, total_threads>>>(tensor);
        cudaDeviceSynchronize();

        glm::fmat3 h_grad;
        cudaMemcpy(&h_grad, d_grad, sizeof(glm::fmat3), cudaMemcpyDeviceToHost);

        std::cout << "\nfmat3 test:\n";
        std::cout << "Expected: all elements = " << total_threads * 1.0f
                  << "\n";
        std::cout << "Got:\n";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_grad[i][j] << " ";
            }
            std::cout << "\n";
        }

        cudaFree(d_grad);
    }

    return 0;
}