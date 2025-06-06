#include <initializer_list>
#include <iostream>
#include <string>

#include <glm/glm.hpp>

template <class T> T *create_device_ptr(const T &h_val, const size_t n) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);
    cudaMemcpy(d_ptr, &h_val, sizeof(T) * n, cudaMemcpyHostToDevice);
    return d_ptr;
}

template <class T> T *create_device_ptr() {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T));
    return d_ptr;
}

template <class T> T *create_device_ptr(const size_t n) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);
    return d_ptr;
}

template <class T>
T *create_device_ptr_with_init(const size_t n, const float init_val) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);

    // Create a temporary host array with the desired value
    T *h_ptr = new T[n];
    for (size_t i = 0; i < n; i++) {
        h_ptr[i] = static_cast<T>(init_val);
    }

    // Copy the initialized array to device
    cudaMemcpy(d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice);
    delete[] h_ptr;

    return d_ptr;
}

template <class T> T *create_device_ptr(std::initializer_list<T> init_list) {
    T *device_ptr;
    cudaMalloc(&device_ptr, sizeof(T) * init_list.size());
    cudaMemcpy(
        device_ptr,
        init_list.begin(),
        sizeof(T) * init_list.size(),
        cudaMemcpyHostToDevice
    );
    return device_ptr;
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

void check_cuda_error() {
    cudaError_t err;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}