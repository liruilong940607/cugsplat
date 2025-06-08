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

    // Sum all elements
    __host__ __device__ T sum() const {
        T result = T(0);
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
            result += data[i];
        }
        return result;
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
template <size_t N> using fvec = vec<float, N>;
using fvec2 = fvec<2>;
using fvec3 = fvec<3>;
using fvec4 = fvec<4>;

template <size_t N> using dvec = vec<double, N>;
using dvec2 = dvec<2>;
using dvec3 = dvec<3>;
using dvec4 = dvec<4>;

template <size_t N> using ivec = vec<int, N>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;

} // namespace tinyrend
