#include <cassert>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_relational.hpp>

#include <torch/torch.h>

#include "glm_tensor.h"
#include "math/cholesky3x3.h"

void test_cholesky() {
    glm::fmat3 A(0.0f);
    A[0][0] = 4.0f;
    A[1][0] = A[0][1] = 2.0f;
    A[1][1] = 5.0f;
    A[2][0] = A[0][2] = 2.0f;
    A[2][1] = A[1][2] = 1.0f;
    A[2][2] = 3.0f;

    auto [L, ok] = gsplat::cholesky(A);
    assert(ok && "cholesky decomposition failed");

    glm::fmat3 LLT = L * glm::transpose(L);
    assert(glm::all(glm::equal(LLT, A, 1e-5f)));
}

void test_forward_substitution() {
    glm::fmat3 L(0.0f);
    L[0][0] = 1.0f;
    L[1][1] = 2.0f;
    L[2][2] = 3.0f;
    glm::fvec3 y(1.0f, 4.0f, 9.0f);

    glm::fvec3 x = gsplat::forward_substitution(L, y);
    glm::fvec3 y_reconstructed = L * x;

    assert(glm::all(glm::equal(y, y_reconstructed, 1e-5f)));
}

void test_backward_substitution() {
    glm::fmat3 L(0.0f);
    L[0][0] = 1.0f;
    L[1][1] = 2.0f;
    L[2][2] = 3.0f;
    glm::fvec3 y(3.0f, 2.0f, 1.0f);

    glm::fvec3 x = gsplat::backward_substitution(L, y);
    glm::fvec3 y_reconstructed = glm::transpose(L) * x;

    assert(glm::all(glm::equal(y, y_reconstructed, 1e-5f)));
}

void test_forward_substitution_vjp() {
    glm::fmat3 L = glm::fmat3(1.0f);
    glm::fvec3 y = glm::fvec3(1.0f);
    glm::fvec3 x = gsplat::forward_substitution(L, y);
    glm::fvec3 v_x = glm::fvec3(1.0f);

    glm::fmat3 v_L = gsplat::forward_substitution_vjp(L, x, v_x);
    assert(v_L[0][0] < 0);
}

void test_cholesky_Winv_y() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = gsplat::cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fvec3 y(1.0f);
    glm::fvec3 x = gsplat::cholesky_Winv_y(L, y);
    assert(glm::all(glm::equal(A * x, y, 1e-5f)));
}

void test_cholesky_Winv() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = gsplat::cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 Winv = gsplat::cholesky_Winv(L);
    assert(glm::all(glm::equal(A * Winv, glm::fmat3(1.0f), 1e-5f)));
}

void test_cholesky_Linv() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = gsplat::cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 Linv = gsplat::cholesky_Linv(L);
    assert(glm::all(glm::equal(L * Linv, glm::fmat3(1.0f), 1e-5f)));
}

void test_cholesky_Linv_vjp() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    glm::fmat3 v_Linv(0.0f); // gradient of L^-1, lower triangular
    v_Linv[0][0] = 1.0f;
    v_Linv[0][1] = 0.3f; v_Linv[1][1] = -0.2f;
    v_Linv[0][2] = 0.5f; v_Linv[1][2] = 0.7f; v_Linv[2][2] = 2.0f;

    auto [L, ok] = gsplat::cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 Linv = gsplat::cholesky_Linv(L);
    glm::fmat3 v_L = gsplat::cholesky_Linv_vjp(L, v_Linv);

    // reference
    torch::Tensor L_torch = glm_to_tensor(L).requires_grad_(true);
    torch::Tensor Linv_torch = torch::linalg_inv(L_torch);
    torch::Tensor v_Linv_torch = glm_to_tensor(v_Linv);
    std::vector<torch::Tensor> grads = torch::autograd::grad(
        {Linv_torch},
        {L_torch},
        {v_Linv_torch},
        /*retain_graph=*/false,
        /*create_graph=*/false
    );
    glm::fmat3 v_L_reference = tensor_to_glm<glm::fmat3>(grads[0]);
    assert(glm::all(glm::equal(v_L, v_L_reference, 1e-5f)));
}

void test_cholesky_vjp() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    glm::fmat3 v_L(0.0f); // gradient of L, lower triangular
    v_L[0][0] = 1.0f;
    v_L[0][1] = 0.3f; v_L[1][1] = -0.2f;
    v_L[0][2] = 0.5f; v_L[1][2] = 0.7f; v_L[2][2] = 2.0f;

    auto [L, ok] = gsplat::cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 v_A = gsplat::cholesky_vjp(L, v_L);

    // reference
    torch::Tensor A_torch = glm_to_tensor(A).requires_grad_(true);
    torch::Tensor L_torch = torch::linalg_cholesky(A_torch);
    torch::Tensor v_L_torch = glm_to_tensor(v_L);
    std::vector<torch::Tensor> grads = torch::autograd::grad(
        {L_torch},
        {A_torch},
        {v_L_torch},
        /*retain_graph=*/false,
        /*create_graph=*/false
    );
    glm::fmat3 v_A_reference = tensor_to_glm<glm::fmat3>(grads[0]);
    assert(glm::all(glm::equal(v_A, v_A_reference, 1e-5f)));
}

int main() {
    test_cholesky();
    test_cholesky_vjp();

    test_forward_substitution();
    test_forward_substitution_vjp();

    test_backward_substitution();

    test_cholesky_Linv();
    test_cholesky_Linv_vjp();

    test_cholesky_Winv_y();
    test_cholesky_Winv();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}