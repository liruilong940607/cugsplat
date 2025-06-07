#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include "tinyrend/rasterization/kernel.cuh"
#include "tinyrend/rasterization/primitives/image_gaussian.h"
// #include "tinyrend/rasterization/primitives/image_triangle.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace tinyrend::rasterization;

void save_png(float *buffer, int width, int height, const char *filename) {
    // Convert float buffer to unsigned char buffer
    unsigned char *image_data = new unsigned char[width * height];

    // Normalize and convert float values to 0-255 range
    for (int i = 0; i < width * height; i++) {
        // Clamp values between 0 and 1
        float value = std::max(0.0f, std::min(1.0f, buffer[i]));
        // Convert to 0-255 range
        image_data[i] = static_cast<unsigned char>(value * 255.0f);
    }

    // Save as PNG
    stbi_write_png(filename, width, height, 1, image_data, width);

    // Clean up
    delete[] image_data;
}

auto test_rasterization() -> int {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    const int n_primitives = 2;

    // Create Some Image Gaussians on GPU
    glm::fvec2 *h_mu = new glm::fvec2[n_primitives];
    for (int i = 0; i < n_primitives; i++) {
        h_mu[i] = glm::fvec2(i, i) * 4.0f + 6.0f;
    }
    glm::fvec3 *h_conics = new glm::fvec3[n_primitives];
    for (int i = 0; i < n_primitives; i++) {
        h_conics[i] = glm::fvec3(0.25f, 0.0f, 0.25f);
    }
    float *h_features = new float[n_primitives * 3];
    for (int i = 0; i < n_primitives; i++) {
        h_features[i * 3 + 0] = i;
        h_features[i * 3 + 1] = i;
        h_features[i * 3 + 2] = i;
    }

    // Create Some Image Gaussians on GPU
    glm::fvec2 *d_mu = create_device_ptr(h_mu[0], n_primitives);
    glm::fvec3 *d_conics = create_device_ptr(h_conics[0], n_primitives);
    float *d_features = create_device_ptr(h_features[0], n_primitives * 3);

    ImageGaussians<3> primitives{};
    primitives.mu = d_mu;
    primitives.conics = d_conics;
    primitives.features = d_features;

    // Create isect info on GPU
    uint32_t isect_primitive_ids_host[n_primitives] = {0, 1};
    uint32_t isect_prefix_sum_per_tile_host[1] = {2};
    uint32_t *isect_primitive_ids;
    cudaMalloc(&isect_primitive_ids, sizeof(uint32_t) * n_primitives);
    cudaMemcpy(
        isect_primitive_ids,
        isect_primitive_ids_host,
        sizeof(uint32_t) * n_primitives,
        cudaMemcpyHostToDevice
    );
    uint32_t *isect_prefix_sum_per_tile;
    cudaMalloc(&isect_prefix_sum_per_tile, sizeof(uint32_t) * 1);
    cudaMemcpy(
        isect_prefix_sum_per_tile,
        isect_prefix_sum_per_tile_host,
        sizeof(uint32_t) * 1,
        cudaMemcpyHostToDevice
    );

    // image size
    const uint32_t image_h = 16;
    const uint32_t image_w = 16;

    // Create buffer for alpha values
    float *buffer_alpha;
    cudaMalloc(&buffer_alpha, sizeof(float) * image_h * image_w);

    // Create buffer for features
    cudaMalloc(&primitives.buffer_features, sizeof(float) * image_h * image_w * 3);

    // launch rasterization kernel
    dim3 threads(16, 16, 1);
    dim3 grid(1, 1, 1);
    size_t shmem_size = ImageGaussians<3>::shmem_size_per_primitive() * 16 * 16;
    rasterization<<<grid, threads, shmem_size>>>(
        primitives,
        image_h,
        image_w,
        isect_primitive_ids,
        isect_prefix_sum_per_tile,
        buffer_alpha,
        nullptr // buffer_last_primitive_id
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // copy buffer_alpha back to host
    float *buffer_alpha_host = new float[image_h * image_w];
    cudaMemcpy(
        buffer_alpha_host,
        buffer_alpha,
        sizeof(float) * image_h * image_w,
        cudaMemcpyDeviceToHost
    );

    // print buffer_alpha
    for (int i = 0; i < image_h; i++) {
        for (int j = 0; j < image_w; j++) {
            printf("%f ", buffer_alpha_host[i * image_w + j]);
        }
        printf("\n");
    }

    // save buffer_alpha_host into a png file
    save_png(buffer_alpha_host, image_w, image_h, "buffer_alpha.png");

    return 0;
}

// auto test_rasterization2() -> int {
//     cudaError_t err = cudaSetDevice(0);
//     if (err != cudaSuccess) {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//     }

//     const int n_primitives = 2;

//     // Create Some Image Triangles on GPU
//     glm::fvec2 *h_v0 = new glm::fvec2[n_primitives];
//     for (int i = 0; i < n_primitives; i++) {
//         h_v0[i] = glm::fvec2(i, i) * 4.0f + 0.0f;
//     }
//     glm::fvec2 *h_v1 = new glm::fvec2[n_primitives];
//     for (int i = 0; i < n_primitives; i++) {
//         h_v1[i] = glm::fvec2(i + 4, i) * 4.0f + 0.0f;
//     }
//     glm::fvec2 *h_v2 = new glm::fvec2[n_primitives];
//     for (int i = 0; i < n_primitives; i++) {
//         h_v2[i] = glm::fvec2(i, i + 4) * 4.0f + 0.0f;
//     }

//     // Create Some Image Triangles on GPU
//     glm::fvec2 *d_v0 = create_device_ptr(h_v0[0], n_primitives);
//     glm::fvec2 *d_v1 = create_device_ptr(h_v1[0], n_primitives);
//     glm::fvec2 *d_v2 = create_device_ptr(h_v2[0], n_primitives);

//     ImageTriangles primitives{};
//     primitives.v0 = d_v0;
//     primitives.v1 = d_v1;
//     primitives.v2 = d_v2;

//     // Create isect info on GPU
//     uint32_t isect_primitive_ids_host[n_primitives] = {0, 1};
//     uint32_t isect_prefix_sum_per_tile_host[1] = {2};
//     uint32_t *isect_primitive_ids;
//     cudaMalloc(&isect_primitive_ids, sizeof(uint32_t) * n_primitives);
//     cudaMemcpy(
//         isect_primitive_ids,
//         isect_primitive_ids_host,
//         sizeof(uint32_t) * n_primitives,
//         cudaMemcpyHostToDevice
//     );
//     uint32_t *isect_prefix_sum_per_tile;
//     cudaMalloc(&isect_prefix_sum_per_tile, sizeof(uint32_t) * 1);
//     cudaMemcpy(
//         isect_prefix_sum_per_tile,
//         isect_prefix_sum_per_tile_host,
//         sizeof(uint32_t) * 1,
//         cudaMemcpyHostToDevice
//     );

//     // image size
//     const uint32_t image_h = 16;
//     const uint32_t image_w = 16;

//     // Create buffer for alpha values
//     float *buffer_alpha;
//     cudaMalloc(&buffer_alpha, sizeof(float) * image_h * image_w);

//     // launch rasterization kernel
//     dim3 threads(16, 16, 1);
//     dim3 grid(1, 1, 1);
//     size_t shmem_size = ImageTriangles::shmem_size_per_primitive() * 16 * 16;
//     rasterization<<<grid, threads, shmem_size>>>(
//         primitives,
//         image_h,
//         image_w,
//         isect_primitive_ids,
//         isect_prefix_sum_per_tile,
//         buffer_alpha,
//         nullptr // buffer_last_primitive_id
//     );

//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//     }
//     err = cudaDeviceSynchronize();
//     if (err != cudaSuccess) {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//     }

//     // copy buffer_alpha back to host
//     float *buffer_alpha_host = new float[image_h * image_w];
//     cudaMemcpy(
//         buffer_alpha_host,
//         buffer_alpha,
//         sizeof(float) * image_h * image_w,
//         cudaMemcpyDeviceToHost
//     );

//     // print buffer_alpha
//     for (int i = 0; i < image_h; i++) {
//         for (int j = 0; j < image_w; j++) {
//             printf("%f ", buffer_alpha_host[i * image_w + j]);
//         }
//         printf("\n");
//     }

//     // save buffer_alpha_host into a png file
//     save_png(buffer_alpha_host, image_w, image_h, "buffer_alpha.png");

//     return 0;
// }

auto main() -> int {
    int fails = 0;
    fails += test_rasterization();

    if (fails == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d tests failed!\n", fails);
    }

    return fails;
}