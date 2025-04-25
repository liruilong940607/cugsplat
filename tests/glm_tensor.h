#pragma once
#include <cassert>
#include <glm/glm.hpp>
#include <torch/torch.h>
#include <type_traits>

// Type trait to detect whether a type is a GLM matrix
template <typename T> struct is_glm_vec : std::false_type {};
template <glm::length_t L, typename T, glm::qualifier Q>
struct is_glm_vec<glm::vec<L, T, Q>> : std::true_type {};

template <typename T> struct is_glm_mat : std::false_type {};
template <glm::length_t C, glm::length_t R, typename T, glm::qualifier Q>
struct is_glm_mat<glm::mat<C, R, T, Q>> : std::true_type {};

// ---- glm_to_tensor: Matrix version ----
template <typename MatT, std::enable_if_t<is_glm_mat<MatT>::value, int> = 0>
torch::Tensor glm_to_tensor(const MatT &mat) {
    constexpr int Cols = MatT::length();           // Columns
    constexpr int Rows = MatT::col_type::length(); // Rows
    float data[Rows * Cols];

    for (int c = 0; c < Cols; ++c)
        for (int r = 0; r < Rows; ++r)
            data[r * Cols + c] = mat[c][r];

    return torch::from_blob(data, {Rows, Cols}, torch::kFloat32).clone();
}

// ---- tensor_to_glm: Matrix version ----
template <typename MatT, std::enable_if_t<is_glm_mat<MatT>::value, int> = 0>
MatT tensor_to_glm(const torch::Tensor &tensor) {
    constexpr int Cols = MatT::length();
    constexpr int Rows = MatT::col_type::length();
    assert(
        tensor.dim() == 2 && tensor.size(0) == Rows && tensor.size(1) == Cols
    );

    MatT mat;
    auto acc = tensor.accessor<float, 2>();
    for (int c = 0; c < Cols; ++c)
        for (int r = 0; r < Rows; ++r)
            mat[c][r] = acc[r][c];

    return mat;
}

// ---- glm_to_tensor: Vector version ----
template <typename VecT, std::enable_if_t<is_glm_vec<VecT>::value, int> = 0>
torch::Tensor glm_to_tensor(const VecT &vec) {
    constexpr int D = VecT::length();
    float data[D];
    for (int i = 0; i < D; ++i)
        data[i] = vec[i];
    return torch::from_blob(data, {D}, torch::kFloat32).clone();
}

// ---- tensor_to_glm: Vector version ----
template <typename VecT, std::enable_if_t<is_glm_vec<VecT>::value, int> = 0>
VecT tensor_to_glm(const torch::Tensor &tensor) {
    constexpr int D = VecT::length();
    assert(tensor.dim() == 1 && tensor.size(0) == D);

    VecT vec;
    auto acc = tensor.accessor<float, 1>();
    for (int i = 0; i < D; ++i)
        vec[i] = acc[i];
    return vec;
}