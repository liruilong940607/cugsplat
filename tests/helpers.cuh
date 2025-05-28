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