#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>

#include "helpers.cuh"
#include <tinyrend/rasterization.cuh>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace tinyrend;

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

struct ImageGaussians {
    /*
    A collection of 2D Gaussian primitives.
    */

    // Pointers to the device memory (shared across all threads)
    glm::fvec2 *mu;     // [N, 2]
    glm::fvec3 *conics; // [N, 3]

    // Per-thread data
    uint32_t _image_id;
    uint32_t _pixel_x;
    uint32_t _pixel_y;
    void *_shmem_ptr;
    uint32_t _shmem_n_primitives;

    __device__ bool initialize(
        uint32_t image_id,
        uint32_t pixel_x,
        uint32_t pixel_y,
        void *shmem_ptr,
        uint32_t shmem_n_primitives
    ) {
        _image_id = image_id;
        _pixel_x = pixel_x;
        _pixel_y = pixel_y;
        _shmem_ptr = shmem_ptr;
        _shmem_n_primitives = shmem_n_primitives;
        return true;
    }

    __device__ void load_to_shared_memory(uint32_t shmem_id, uint32_t global_id) {
        glm::fvec2 *shmem_ptr_mu = reinterpret_cast<glm::fvec2 *>(_shmem_ptr);
        glm::fvec3 *shmem_ptr_conics =
            reinterpret_cast<glm::fvec3 *>(&shmem_ptr_mu[_shmem_n_primitives]);
        shmem_ptr_mu[shmem_id] = mu[global_id];
        shmem_ptr_conics[shmem_id] = conics[global_id];
    }

    __device__ float get_light_attenuation(uint32_t shmem_id) {
        glm::fvec2 *shmem_ptr_mu = reinterpret_cast<glm::fvec2 *>(_shmem_ptr);
        glm::fvec3 *shmem_ptr_conics =
            reinterpret_cast<glm::fvec3 *>(&shmem_ptr_mu[_shmem_n_primitives]);
        auto const mu = shmem_ptr_mu[shmem_id];
        auto const conic = shmem_ptr_conics[shmem_id];

        auto const dx = _pixel_x - mu.x;
        auto const dy = _pixel_y - mu.y;
        auto const sigma =
            0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
        return exp(-sigma);
    }
};

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

    // Create Some Image Gaussians on GPU
    glm::fvec2 *d_mu = create_device_ptr(h_mu[0], n_primitives);
    glm::fvec3 *d_conics = create_device_ptr(h_conics[0], n_primitives);

    ImageGaussians primitives{};
    primitives.mu = d_mu;
    primitives.conics = d_conics;

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

    // launch rasterization kernel
    dim3 threads(16, 16, 1);
    dim3 grid(1, 1, 1);
    size_t shmem_size = (sizeof(glm::fvec2) + sizeof(glm::fvec3)) * 16 * 16;
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