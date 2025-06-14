#pragma once

#include "vec.h"
#include <array>
#include <cstddef>

namespace tinyrend {

template <typename T, size_t Rows, size_t Cols> struct alignas(T) mat {
    T data[Cols][Rows]; // Column-major storage: [column][row]

    // Default constructor
    mat() = default;

    // Initialize from pointer (column-major)
    __host__ __device__ explicit mat(const T *ptr) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] = ptr[i + j * Rows];
            }
        }
    }

    // Access operators [col]
    __host__ __device__ T &operator[](size_t col) { return data[col]; }
    __host__ __device__ const T &operator[](size_t col) const { return data[col]; }

    // Access operators [col][row]
    __host__ __device__ T &operator[](size_t col, size_t row) { return data[col][row]; }
    __host__ __device__ const T &operator[](size_t col, size_t row) const {
        return data[col][row];
    }

    // Matrix-Matrix operations
    __host__ __device__ mat<T, Rows, Cols> operator+(const mat<T, Rows, Cols> &other
    ) const {
        mat<T, Rows, Cols> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = data[j][i] + other(i, j);
            }
        }
        return result;
    }

    template <size_t OtherCols>
    __host__ __device__ mat<T, Rows, OtherCols>
    operator*(const mat<T, Cols, OtherCols> &other) const {
        mat<T, Rows, OtherCols> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < OtherCols; ++j) {
                result(i, j) = T(0);
#pragma unroll
                for (size_t k = 0; k < Cols; ++k) {
                    result(i, j) += data[k][i] * other(i, k);
                }
            }
        }
        return result;
    }

    __host__ __device__ mat<T, Rows, Cols> operator-(const mat<T, Rows, Cols> &other
    ) const {
        mat<T, Rows, Cols> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = data[j][i] - other(i, j);
            }
        }
        return result;
    }

    // Matrix-Scalar operations
    __host__ __device__ mat<T, Rows, Cols> operator*(T scalar) const {
        mat<T, Rows, Cols> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = data[j][i] * scalar;
            }
        }
        return result;
    }

    __host__ __device__ mat<T, Rows, Cols> operator/(T scalar) const {
        mat<T, Rows, Cols> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = data[j][i] / scalar;
            }
        }
        return result;
    }

    // Matrix-Vector multiplication
    template <size_t N>
    __host__ __device__ vec<T, Rows> operator*(const vec<T, N> &v) const {
        static_assert(N == Cols, "Vector dimension must match matrix columns");
        vec<T, Rows> result;
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
            result[i] = T(0);
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                result[i] += data[j][i] * v[j];
            }
        }
        return result;
    }

    // Compound assignment operators
    __host__ __device__ mat<T, Rows, Cols> &operator+=(const mat<T, Rows, Cols> &other
    ) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] += other(i, j);
            }
        }
        return *this;
    }

    __host__ __device__ mat<T, Rows, Cols> &operator-=(const mat<T, Rows, Cols> &other
    ) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] -= other(i, j);
            }
        }
        return *this;
    }

    __host__ __device__ mat<T, Rows, Cols> &operator*=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] *= scalar;
            }
        }
        return *this;
    }

    __host__ __device__ mat<T, Rows, Cols> &operator/=(T scalar) {
#pragma unroll
        for (size_t i = 0; i < Rows; ++i) {
#pragma unroll
            for (size_t j = 0; j < Cols; ++j) {
                data[j][i] /= scalar;
            }
        }
        return *this;
    }
};

// Common type aliases
template <size_t Rows, size_t Cols> using fmat = mat<float, Rows, Cols>;
using fmat2x2 = fmat<2, 2>;
using fmat3x3 = fmat<3, 3>;
using fmat4x4 = fmat<4, 4>;
using fmat3x4 = fmat<3, 4>;
using fmat4x3 = fmat<4, 3>;

template <size_t Rows, size_t Cols> using dmat = mat<double, Rows, Cols>;
using dmat2x2 = dmat<2, 2>;
using dmat3x3 = dmat<3, 3>;
using dmat4x4 = dmat<4, 4>;
using dmat3x4 = dmat<3, 4>;
using dmat4x3 = dmat<4, 3>;

} // namespace tinyrend