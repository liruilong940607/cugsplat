#pragma once

#include <initializer_list>
#include <iostream>
#include <string>

#include <glm/glm.hpp>

// Basic device pointer creation with optional initialization
template <class T> T *create_device_ptr(const size_t n, const T &value) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);

    // Create temporary host array and initialize
    T *h_ptr = new T[n];
    for (size_t i = 0; i < n; i++) {
        h_ptr[i] = value;
    }

    cudaMemcpy(d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice);
    delete[] h_ptr;
    return d_ptr;
}

// Device pointer creation without initialization
template <class T> T *create_device_ptr(const size_t n) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);
    return d_ptr;
}

// Device pointer creation from initializer list
template <class T> T *create_device_ptr(std::initializer_list<T> init_list) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * init_list.size());
    cudaMemcpy(
        d_ptr, init_list.begin(), sizeof(T) * init_list.size(), cudaMemcpyHostToDevice
    );
    return d_ptr;
}

// Host to device pointer conversion
template <class T> T *host_ptr_to_device_ptr(const T *h_ptr, const size_t n) {
    T *d_ptr;
    cudaMalloc(&d_ptr, sizeof(T) * n);
    cudaMemcpy(d_ptr, h_ptr, sizeof(T) * n, cudaMemcpyHostToDevice);
    return d_ptr;
}

// Device to host pointer conversion
template <class T> T *device_ptr_to_host_ptr(const T *d_ptr, const size_t n) {
    T *h_ptr = new T[n];
    cudaMemcpy(h_ptr, d_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
    return h_ptr;
}

// Helper to print the first element of the device pointer
template <class T> void print_device_ptr(const T *d_ptr, const std::string &name) {
    T h_val;
    cudaMemcpy(&h_val, d_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << name << ": " << h_val << std::endl;
}

// Specialization for glm::fvec2
template <>
void print_device_ptr<glm::fvec2>(const glm::fvec2 *d_ptr, const std::string &name) {
    glm::fvec2 h_val;
    cudaMemcpy(&h_val, d_ptr, sizeof(glm::fvec2), cudaMemcpyDeviceToHost);
    std::cout << name << ": " << h_val.x << ", " << h_val.y << std::endl;
}

// Specialization for glm::fvec3
template <>
void print_device_ptr<glm::fvec3>(const glm::fvec3 *d_ptr, const std::string &name) {
    glm::fvec3 h_val;
    cudaMemcpy(&h_val, d_ptr, sizeof(glm::fvec3), cudaMemcpyDeviceToHost);
    std::cout << name << ": " << h_val.x << ", " << h_val.y << ", " << h_val.z
              << std::endl;
}

// CUDA error checking
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