#pragma once

#include <array>
#include <cstddef>

namespace tinyrend {

template <typename T, size_t N> struct alignas(T) vec {
    T data[N]; // Use raw array instead of std::array to match glm's layout

    // Default constructor
    __host__ __device__ vec() = default;

    // Initialize from pointer
    __host__ __device__ explicit vec(const T *ptr) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] = ptr[i];
        }
    }

    // Initialize from initializer list
    template <typename... Args>
    __host__ __device__ vec(Args... args) : data{static_cast<T>(args)...} {
        static_assert(
            sizeof...(Args) == N, "Number of arguments must match vector size"
        );
    }

    // Access operators
    __host__ __device__ T &operator[](size_t i) { return data[i]; }
    __host__ __device__ const T &operator[](size_t i) const { return data[i]; }

    // Pointer casting operators
    __host__ __device__ operator T *() { return data; }
    __host__ __device__ operator const T *() const { return data; }

    // Vector-Vector operations
    __host__ __device__ vec<T, N> operator+(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    __host__ __device__ vec<T, N> operator-(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    __host__ __device__ vec<T, N> operator*(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] * other[i];
        }
        return result;
    }

    __host__ __device__ vec<T, N> operator/(const vec<T, N> &other) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] / other[i];
        }
        return result;
    }

    // Vector-Scalar operations
    __host__ __device__ vec<T, N> operator+(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] + scalar;
        }
        return result;
    }

    __host__ __device__ vec<T, N> operator-(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] - scalar;
        }
        return result;
    }

    __host__ __device__ vec<T, N> operator*(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    __host__ __device__ vec<T, N> operator/(T scalar) const {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] / scalar;
        }
        return result;
    }

    // Scalar-Vector operations (friend functions)
    __host__ __device__ friend vec<T, N> operator+(T scalar, const vec<T, N> &v) {
        return v + scalar;
    }

    __host__ __device__ friend vec<T, N> operator-(T scalar, const vec<T, N> &v) {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = scalar - v[i];
        }
        return result;
    }

    __host__ __device__ friend vec<T, N> operator*(T scalar, const vec<T, N> &v) {
        return v * scalar;
    }

    __host__ __device__ friend vec<T, N> operator/(T scalar, const vec<T, N> &v) {
        vec<T, N> result;
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result[i] = scalar / v[i];
        }
        return result;
    }

    // Compound assignment operators
    __host__ __device__ vec<T, N> &operator+=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator-=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] -= other[i];
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator*=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] *= other[i];
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator/=(const vec<T, N> &other) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] /= other[i];
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator+=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] += scalar;
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator-=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] -= scalar;
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator*=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    __host__ __device__ vec<T, N> &operator/=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            data[i] /= scalar;
        }
        return *this;
    }
};

// Common type aliases
template <typename T> using vec2 = vec<T, 2>;

template <typename T> using vec3 = vec<T, 3>;

template <typename T> using vec4 = vec<T, 4>;

// Common type aliases for float
using fvec2 = vec2<float>;
using fvec3 = vec3<float>;
using fvec4 = vec4<float>;

// Common type aliases for double
using dvec2 = vec2<double>;
using dvec3 = vec3<double>;
using dvec4 = vec4<double>;

// Common type aliases for int
using ivec2 = vec2<int>;
using ivec3 = vec3<int>;
using ivec4 = vec4<int>;

} // namespace tinyrend
