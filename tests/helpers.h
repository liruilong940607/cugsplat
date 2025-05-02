#pragma once

#include <cmath>
#include <glm/glm.hpp>

// Helper function to compute numerical gradient using central differences
template <typename T, typename F>
T numerical_gradient(const T &x, F f, float eps = 1e-4f) {
    T grad;
    for (int i = 0; i < x.length(); ++i) {
        T x_plus = x;
        T x_minus = x;
        x_plus[i] += eps;
        x_minus[i] -= eps;
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0f * eps);
    }
    return grad;
}

// Helper function to check if two vectors are close within absolute and
// relative tolerances
template <typename T>
bool is_close(const T &a, const T &b, float atol = 1e-2f, float rtol = 1e-2f) {
    for (int i = 0; i < a.length(); ++i) {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) {
            return false;
        }
    }
    return true;
}

template <>
bool is_close(const float &a, const float &b, float atol, float rtol) {
    return std::abs(a - b) < atol + rtol * std::abs(b);
}