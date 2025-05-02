#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "core/math.h"

using namespace gsplat::math;

int test_rsqrtf() {
    int fails = 0;

    // Test case 1: Normal values
    {
        auto const x = 4.0f;
        auto const y = rsqrtf(x);
        auto const y_expected = 0.5f;
        if (!is_close(y, y_expected)) {
            printf("\n=== Testing rsqrtf ===\n");
            printf("\n[FAIL] Test 1: Normal values\n");
            printf("  Input: %f\n", x);
            printf("  Output: %f\n", y);
            printf("  Expected: %f\n", y_expected);
            printf("  Error: %f\n", std::abs(y - y_expected));
            fails += 1;
        }
    }

    // Test case 2: Small values
    {
        auto const x = 1e-6f;
        auto const y = rsqrtf(x);
        auto const y_expected = 1.0f / std::sqrt(x);
        if (!is_close(y, y_expected)) {
            printf("\n[FAIL] Test 2: Small values\n");
            printf("  Input: %f\n", x);
            printf("  Output: %f\n", y);
            printf("  Expected: %f\n", y_expected);
            printf("  Error: %f\n", std::abs(y - y_expected));
            fails += 1;
        }
    }

    return fails;
}

int test_numerically_stable_norm2() {
    int fails = 0;

    // Test case 1: Normal values
    {
        auto const x = 3.0f;
        auto const y = 4.0f;
        auto const norm = numerically_stable_norm2(x, y);
        auto const norm_expected = 5.0f;
        if (!is_close(norm, norm_expected)) {
            printf("\n=== Testing numerically_stable_norm2 ===\n");
            printf("\n[FAIL] Test 1: Normal values\n");
            printf("  Input: (%f, %f)\n", x, y);
            printf("  Output: %f\n", norm);
            printf("  Expected: %f\n", norm_expected);
            printf("  Error: %f\n", std::abs(norm - norm_expected));
            fails += 1;
        }
    }

    // Test case 2: Large difference in magnitude
    {
        auto const x = 1e-6f;
        auto const y = 1e6f;
        auto const norm = numerically_stable_norm2(x, y);
        auto const norm_expected = y;
        if (!is_close(norm, norm_expected)) {
            printf("\n[FAIL] Test 2: Large difference in magnitude\n");
            printf("  Input: (%f, %f)\n", x, y);
            printf("  Output: %f\n", norm);
            printf("  Expected: %f\n", norm_expected);
            printf("  Error: %f\n", std::abs(norm - norm_expected));
            fails += 1;
        }
    }

    return fails;
}

int test_eval_poly_horner() {
    int fails = 0;

    // Test case 1: Quadratic polynomial
    {
        std::array<float, 3> poly = {1.0f, 2.0f, 3.0f}; // 1 + 2x + 3x^2
        auto const x = 2.0f;
        auto const y = eval_poly_horner(poly, x);
        auto const y_expected = 17.0f;
        if (!is_close(y, y_expected)) {
            printf("\n=== Testing eval_poly_horner ===\n");
            printf("\n[FAIL] Test 1: Quadratic polynomial\n");
            printf("  Input: %f\n", x);
            printf("  Output: %f\n", y);
            printf("  Expected: %f\n", y_expected);
            printf("  Error: %f\n", std::abs(y - y_expected));
            fails += 1;
        }
    }

    // Test case 2: Constant polynomial
    {
        std::array<float, 1> poly = {42.0f};
        auto const x = 1.0f;
        auto const y = eval_poly_horner(poly, x);
        auto const y_expected = 42.0f;
        if (!is_close(y, y_expected)) {
            printf("\n[FAIL] Test 2: Constant polynomial\n");
            printf("  Input: %f\n", x);
            printf("  Output: %f\n", y);
            printf("  Expected: %f\n", y_expected);
            printf("  Error: %f\n", std::abs(y - y_expected));
            fails += 1;
        }
    }

    return fails;
}

int test_safe_normalize() {
    int fails = 0;

    // Test case 1: Normal vector
    {
        auto const v = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const n = safe_normalize(v);
        auto const length = glm::length(n);
        if (!is_close(length, 1.0f)) {
            printf("\n=== Testing safe_normalize ===\n");
            printf("\n[FAIL] Test 1: Normal vector\n");
            printf("  Input: %s\n", glm::to_string(v).c_str());
            printf("  Output: %s\n", glm::to_string(n).c_str());
            printf("  Length: %f\n", length);
            printf("  Expected length: 1.0\n");
            fails += 1;
        }
    }

    // Test case 2: Zero vector
    {
        auto const v = glm::fvec3(0.0f);
        auto const n = safe_normalize(v);
        if (n != v) {
            printf("\n[FAIL] Test 2: Zero vector\n");
            printf("  Input: %s\n", glm::to_string(v).c_str());
            printf("  Output: %s\n", glm::to_string(n).c_str());
            printf("  Expected: %s\n", glm::to_string(v).c_str());
            fails += 1;
        }
    }

    return fails;
}

int test_safe_normalize_vjp() {
    int fails = 0;

    // Test case 1: Normal vector
    {
        auto const x = glm::fvec3(1.0f, 2.0f, 3.0f);
        auto const v_out = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const grad = safe_normalize_vjp(x, v_out);

        // Compute numerical gradient
        auto const grad_num = numerical_gradient(x, [&](const glm::fvec3 &x) {
            auto const n = safe_normalize(x);
            return glm::dot(v_out, n);
        });

        if (!is_close(grad, grad_num)) {
            printf("\n=== Testing safe_normalize_vjp ===\n");
            printf("\n[FAIL] Test 1: Normal vector\n");
            printf("  Input: %s\n", glm::to_string(x).c_str());
            printf("  v_out: %s\n", glm::to_string(v_out).c_str());
            printf("  Analytical gradient: %s\n", glm::to_string(grad).c_str());
            printf(
                "  Numerical gradient: %s\n", glm::to_string(grad_num).c_str()
            );
            printf("  Error: %f\n", glm::length(grad - grad_num));
            fails += 1;
        }
    }

    // Test case 2: Zero vector
    {
        auto const x = glm::fvec3(0.0f);
        auto const v_out = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const grad = safe_normalize_vjp(x, v_out);
        if (grad != v_out) {
            printf("\n[FAIL] Test 2: Zero vector\n");
            printf("  Input: %s\n", glm::to_string(x).c_str());
            printf("  v_out: %s\n", glm::to_string(v_out).c_str());
            printf("  Output: %s\n", glm::to_string(grad).c_str());
            printf("  Expected: %s\n", glm::to_string(v_out).c_str());
            fails += 1;
        }
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_rsqrtf();
    fails += test_numerically_stable_norm2();
    fails += test_eval_poly_horner();
    fails += test_safe_normalize();
    fails += test_safe_normalize_vjp();

    if (fails > 0) {
        printf("[math/utils.cpp] %d tests failed!\n", fails);
    } else {
        printf("[math/utils.cpp] All tests passed!\n");
    }

    return fails;
}