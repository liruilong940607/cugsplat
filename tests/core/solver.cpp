#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "curend/core/solver.h"

using namespace curend::solver;

int test_newton_1d() {
    int fails = 0;

    // Test case 1: Simple quadratic equation x^2 - 4 = 0
    {
        auto const func = [](const float &x) -> std::pair<float, float> {
            return {x * x - 4.0f, 2.0f * x};
        };
        auto const result = newton_1d<10>(func, 3.0f);
        if (!result.converged || !is_close(result.x, 2.0f)) {
            printf("\n=== Testing newton_1d ===\n");
            printf("\n[FAIL] Test 1: Quadratic equation\n");
            printf("  Root: %f\n", result.x);
            printf("  Expected: 2.0\n");
            printf("  Converged: %d\n", result.converged);
            fails += 1;
        }
    }

    // Test case 2: No convergence
    {
        auto const func = [](const float &x) -> std::pair<float, float> {
            return {1.0f, 0.0f}; // f(x) = 1, f'(x) = 0
        };
        auto const result = newton_1d<10>(func, 1.0f);
        if (result.converged) {
            printf("\n[FAIL] Test 2: No convergence\n");
            printf("  Root: %f\n", result.x);
            printf("  Converged: %d\n", result.converged);
            printf("  Expected: not converged\n");
            fails += 1;
        }
    }

    return fails;
}

int test_newton_2d() {
    int fails = 0;

    // Test case 1: Simple system x^2 + y^2 = 5, x + y = 3
    {
        auto const func = [](const glm::fvec2 &xy
                          ) -> std::pair<glm::fvec2, glm::fmat2> {
            auto const x = xy[0];
            auto const y = xy[1];
            return {
                {x * x + y * y - 5.0f, x + y - 3.0f},
                {{2.0f * x, 2.0f * y}, {1.0f, 1.0f}}
            };
        };
        auto const result = newton_2d<10>(func, {1.5f, 1.0f});
        if (!result.converged || !is_close(result.x, glm::fvec2(2.0f, 1.0f))) {
            printf("\n=== Testing newton_2d ===\n");
            printf("\n[FAIL] Test 1: Simple system\n");
            printf("  Root: %s\n", glm::to_string(result.x).c_str());
            printf("  Expected: (2.0, 1.0)\n");
            printf("  Converged: %d\n", result.converged);
            fails += 1;
        }
    }

    return fails;
}

int test_linear_minimal_positive() {
    int fails = 0;

    // Test case 1: Simple linear equation
    {
        auto const poly = std::array<float, 2>{-2.0f, 1.0f}; // -2 + x = 0
        auto const root = linear_minimal_positive(poly, 0.0f, -1.0f);
        if (!is_close(root, 2.0f)) {
            printf("\n=== Testing linear_minimal_positive ===\n");
            printf("\n[FAIL] Test 1: Simple linear equation\n");
            printf("  Root: %f\n", root);
            printf("  Expected: 2.0\n");
            fails += 1;
        }
    }

    // Test case 2: No positive root
    {
        auto const poly = std::array<float, 2>{2.0f, 1.0f}; // 2 + x = 0
        auto const root = linear_minimal_positive(poly, 0.0f, -1.0f);
        if (root != -1.0f) {
            printf("\n[FAIL] Test 2: No positive root\n");
            printf("  Root: %f\n", root);
            printf("  Expected: -1.0\n");
            fails += 1;
        }
    }

    return fails;
}

int test_quadratic_minimal_positive() {
    int fails = 0;

    // Test case 1: Simple quadratic equation
    {
        auto const poly = std::array<float, 3>{-3.0f, 2.0f, 1.0f}; // -3 + 2x + x^2 = 0
        auto const root = quadratic_minimal_positive(poly, 0.0f, -1.0f);
        if (!is_close(root, 1.0f)) {
            printf("\n=== Testing quadratic_minimal_positive ===\n");
            printf("\n[FAIL] Test 1: Simple quadratic equation\n");
            printf("  Root: %f\n", root);
            printf("  Expected: 1.0\n");
            fails += 1;
        }
    }

    // Test case 2: No positive root
    {
        auto const poly = std::array<float, 3>{3.0f, 2.0f, 1.0f}; // 3 + 2x + x^2 = 0
        auto const root = quadratic_minimal_positive(poly, 0.0f, -1.0f);
        if (root != -1.0f) {
            printf("\n[FAIL] Test 2: No positive root\n");
            printf("  Root: %f\n", root);
            printf("  Expected: -1.0\n");
            fails += 1;
        }
    }

    return fails;
}

int test_cubic_minimal_positive() {
    int fails = 0;

    // Test case 1: Simple cubic equation
    {
        auto const poly =
            std::array<float, 4>{-6.0f, 11.0f, -6.0f, 1.0f}; // (x-1)(x-2)(x-3) = 0
        auto const root = cubic_minimal_positive(poly, 0.0f, -1.0f);
        if (!is_close(root, 1.0f)) {
            printf("\n=== Testing cubic_minimal_positive ===\n");
            printf("\n[FAIL] Test 1: Simple cubic equation\n");
            printf("  Root: %f\n", root);
            printf("  Expected: 1.0\n");
            fails += 1;
        }
    }

    // Test case 2: No positive root
    {
        auto const poly =
            std::array<float, 4>{6.0f, 11.0f, 6.0f, 1.0f}; // (x+1)(x+2)(x+3) = 0
        auto const root = cubic_minimal_positive(poly, 0.0f, -1.0f);
        if (root != -1.0f) {
            printf("\n[FAIL] Test 2: No positive root\n");
            printf("  Root: %f\n", root);
            printf("  Expected: -1.0\n");
            fails += 1;
        }
    }

    return fails;
}

int test_polyN_minimal_positive_newton() {
    int fails = 0;

    // Test case 1: Quartic equation
    {
        auto const poly = std::array<float, 5>{
            -24.0f, 50.0f, -35.0f, 10.0f, -1.0f
        }; // (x-1)(x-2)(x-3)(x-4) = 0
        auto const root = polyN_minimal_positive_newton<10>(poly, 0.0f, 0.5f, -1.0f);
        if (!is_close(root, 1.0f)) {
            printf("\n=== Testing polyN_minimal_positive_newton ===\n");
            printf("\n[FAIL] Test 1: Quartic equation\n");
            printf("  Root: %f\n", root);
            printf("  Expected: 1.0\n");
            fails += 1;
        }
    }

    // Test case 2: No convergence
    {
        auto const poly =
            std::array<float, 5>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // 1 + x^4 = 0
        auto const root = polyN_minimal_positive_newton<10>(poly, 0.0f, 1.0f, -1.0f);
        if (root != -1.0f) {
            printf("\n[FAIL] Test 2: No convergence\n");
            printf("  Root: %f\n", root);
            printf("  Expected: -1.0\n");
            fails += 1;
        }
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_newton_1d();
    fails += test_newton_2d();
    fails += test_linear_minimal_positive();
    fails += test_quadratic_minimal_positive();
    fails += test_cubic_minimal_positive();
    fails += test_polyN_minimal_positive_newton();

    if (fails > 0) {
        printf("[core/solver.cpp] %d tests failed!\n", fails);
    } else {
        printf("[core/solver.cpp] All tests passed!\n");
    }

    return fails;
}