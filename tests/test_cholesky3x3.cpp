#include <cassert>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/ext/matrix_relational.hpp>

#include "math/cholesky3x3.h"

using namespace gsplat;

void test_cholesky_decomposition() {
    glm::fmat3 A(0.0f);
    A[0][0] = 4.0f;
    A[1][0] = A[0][1] = 2.0f;
    A[1][1] = 5.0f;
    A[2][0] = A[0][2] = 2.0f;
    A[2][1] = A[1][2] = 1.0f;
    A[2][2] = 3.0f;

    auto [L, ok] = cholesky(A);
    assert(ok && "Cholesky decomposition failed");

    glm::fmat3 LLT = L * glm::transpose(L);
    assert(glm::all(glm::equal(LLT, A, 1e-3f)));
    std::cout << "Cholesky decomposition test passed!" << std::endl;
}


int main()
{
    test_cholesky_decomposition();
    return 0;
}
