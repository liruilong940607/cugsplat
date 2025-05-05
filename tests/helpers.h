#pragma once

#include <array>
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
    if constexpr (std::is_same_v<T, glm::fquat>) {
        return std::abs(a.w - b.w) <= atol + rtol * std::abs(b.w) &&
               std::abs(a.x - b.x) <= atol + rtol * std::abs(b.x) &&
               std::abs(a.y - b.y) <= atol + rtol * std::abs(b.y) &&
               std::abs(a.z - b.z) <= atol + rtol * std::abs(b.z);
    } else if constexpr (std::is_same_v<T, float>) {
        return std::abs(a - b) < atol + rtol * std::abs(b);
    } else if constexpr (std::is_same_v<T, glm::fmat2> ||
                         std::is_same_v<T, glm::fmat3> ||
                         std::is_same_v<T, glm::fmat4> ||
                         std::is_same_v<T, glm::fmat2x3> ||
                         std::is_same_v<T, glm::fmat3x2>) {
        for (int i = 0; i < T::length(); ++i) {
            for (int j = 0; j < T::col_type::length(); ++j) {
                if (std::abs(a[i][j] - b[i][j]) > atol + rtol * std::abs(b[i][j])) {
                    return false;
                }
            }
        }
        return true;
    } else if constexpr (std::is_array_v<T>) {
        for (size_t i = 0; i < a.size(); ++i) {
            if (!is_close(a[i], b[i], atol, rtol)) {
                return false;
            }
        }
        return true;
    } else {
        for (int i = 0; i < a.length(); ++i) {
            if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) {
                return false;
            }
        }
        return true;
    }
}