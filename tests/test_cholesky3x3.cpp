#include <cassert>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_relational.hpp>

#include "math/cholesky3x3.h"

using namespace gsplat;

void test_cholesky() {
    glm::fmat3 A(0.0f);
    A[0][0] = 4.0f;
    A[1][0] = A[0][1] = 2.0f;
    A[1][1] = 5.0f;
    A[2][0] = A[0][2] = 2.0f;
    A[2][1] = A[1][2] = 1.0f;
    A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
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

    glm::fvec3 x = forward_substitution(L, y);
    glm::fvec3 y_reconstructed = L * x;

    assert(glm::all(glm::equal(y, y_reconstructed, 1e-5f)));
}

void test_backward_substitution() {
    glm::fmat3 L(0.0f);
    L[0][0] = 1.0f;
    L[1][1] = 2.0f;
    L[2][2] = 3.0f;
    glm::fvec3 y(3.0f, 2.0f, 1.0f);

    glm::fvec3 x = backward_substitution(L, y);
    glm::fvec3 y_reconstructed = glm::transpose(L) * x;

    assert(glm::all(glm::equal(y, y_reconstructed, 1e-5f)));
}

void test_forward_substitution_vjp() {
    glm::fmat3 L = glm::fmat3(1.0f);
    glm::fvec3 y = glm::fvec3(1.0f);
    glm::fvec3 x = forward_substitution(L, y);
    glm::fvec3 v_x = glm::fvec3(1.0f);

    glm::fmat3 v_L = forward_substitution_vjp(L, x, v_x);
    assert(v_L[0][0] < 0);
}

void test_cholesky_Winv_y() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fvec3 y(1.0f);
    glm::fvec3 x = cholesky_Winv_y(L, y);
    assert(glm::all(glm::equal(A * x, y, 1e-5f)));
}

void test_cholesky_Winv() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 Winv = cholesky_Winv(L);
    assert(glm::all(glm::equal(A * Winv, glm::fmat3(1.0f), 1e-5f)));
}

void test_cholesky_Linv() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 Linv = cholesky_Linv(L);
    assert(glm::all(glm::equal(L * Linv, glm::fmat3(1.0f), 1e-5f)));
}

void test_cholesky_Linv_vjp() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 Linv = cholesky_Linv(L);
    glm::fmat3 grad = cholesky_Linv_vjp(L, Linv);
    assert(grad[0][0] < 0);
}

void test_cholesky_vjp() {
    glm::fmat3 A;
    A[0][0] = 4.0f; A[0][1] = 2.0f; A[0][2] = 2.0f;
    A[1][0] = 2.0f; A[1][1] = 5.0f; A[1][2] = 1.0f;
    A[2][0] = 2.0f; A[2][1] = 1.0f; A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
    assert(ok && "cholesky decomposition failed");
    glm::fmat3 grad = cholesky_vjp(L, L);
    assert(grad[0][0] > 0);
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