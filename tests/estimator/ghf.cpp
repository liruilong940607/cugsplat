#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <cmath>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "cugsplat/estimator/ghf.h"

using namespace cugsplat::ghf;

// Helper function to compute Frobenius norm of a matrix
template <typename T> float frobenius_norm(T const &mat) {
    float sum = 0.0f;
    for (int i = 0; i < mat.length(); ++i) {
        for (int j = 0; j < mat[0].length(); ++j) {
            float diff = mat[i][j];
            sum += diff * diff;
        }
    }
    return std::sqrt(sum);
}

// Test function: f(x) = [sin(x0) + x1^2 + x0*x1, x2*exp(-x0^2)]
auto test_function = [](glm::vec3 const &x) -> glm::vec2 {
    return {std::sin(x[0]) + x[1] * x[1] + x[0] * x[1], x[2] * std::exp(-x[0] * x[0])};
};

// Analytical Jacobian of test function
auto analytical_jacobian = [](glm::vec3 const &x) -> glm::mat2x3 {
    return {
        std::cos(x[0]) + x[1],
        2.0f * x[1] + x[0],
        0.0f,
        -2.0f * x[0] * x[2] * std::exp(-x[0] * x[0]),
        0.0f,
        std::exp(-x[0] * x[0])
    };
};

// Analytical Hessian of test function
auto analytical_hessian = [](glm::vec3 const &x) -> std::array<glm::mat3, 2> {
    std::array<glm::mat3, 2> H{};

    // First output dimension Hessian
    H[0] = {-std::sin(x[0]), 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Second output dimension Hessian
    float exp_term = std::exp(-x[0] * x[0]);
    H[1] = {
        2.0f * x[2] * exp_term * (2.0f * x[0] * x[0] - 1.0f),
        0.0f,
        -2.0f * x[0] * exp_term,
        0.0f,
        0.0f,
        0.0f,
        -2.0f * x[0] * exp_term,
        0.0f,
        0.0f
    };

    return H;
};

int test_ghf_jacobian() {
    int fails = 0;

    // Test case 1: Normal values
    {
        glm::vec3 mu{0.1f, 0.2f, -0.3f};
        glm::vec3 sigma_diag{0.001f, 0.001f, 0.001f};

        auto [J_est, H_est] =
            estimate_jacobian_and_hessian<3, 2>(test_function, mu, sigma_diag);
        auto J_anal = analytical_jacobian(mu);

        // Compare estimated and analytical Jacobian
        if (!is_close(J_est, J_anal, 1e-3f, 1e-3f)) {
            printf("\n=== Testing GHF Jacobian ===\n");
            printf("\n[FAIL] Test 1: Normal values\n");
            printf("  Input mu: %s\n", glm::to_string(mu).c_str());
            printf("  Input sigma_diag: %s\n", glm::to_string(sigma_diag).c_str());
            printf("  Estimated Jacobian:\n");
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    printf("    %f", J_est[i][j]);
                }
                printf("\n");
            }
            printf("  Analytical Jacobian:\n");
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 3; ++j) {
                    printf("    %f", J_anal[i][j]);
                }
                printf("\n");
            }
            printf("  Error: %f\n", frobenius_norm(J_est - J_anal));
            fails += 1;
        }
    }

    return fails;
}

int test_ghf_hessian() {
    int fails = 0;

    // Test case 1: Normal values
    {
        glm::vec3 mu{0.1f, 0.2f, -0.3f};
        glm::vec3 std_dev{1e-2f, 1e-2f, 1e-1f};

        auto [J_est, H_est] =
            estimate_jacobian_and_hessian<3, 2>(test_function, mu, std_dev);
        auto H_anal = analytical_hessian(mu);

        // Compare estimated and analytical Hessian
        bool hessian_match = true;
        for (int k = 0; k < 2; ++k) {
            if (!is_close(H_est[k], H_anal[k], 1e-3f, 1e-3f)) {
                hessian_match = false;
                break;
            }
        }

        if (!hessian_match) {
            printf("\n=== Testing GHF Hessian ===\n");
            printf("\n[FAIL] Test 1: Normal values\n");
            printf("  Input mu: %s\n", glm::to_string(mu).c_str());
            printf("  Input std_dev: %s\n", glm::to_string(std_dev).c_str());
            printf("  Estimated Hessian:\n");
            for (int k = 0; k < 2; ++k) {
                printf("  Output dimension %d:\n", k);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        printf("    %f", H_est[k][i][j]);
                    }
                    printf("\n");
                }
            }
            printf("  Analytical Hessian:\n");
            for (int k = 0; k < 2; ++k) {
                printf("  Output dimension %d:\n", k);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        printf("    %f", H_anal[k][i][j]);
                    }
                    printf("\n");
                }
            }
            printf(
                "  Error: %f\n",
                frobenius_norm(H_est[0] - H_anal[0]) +
                    frobenius_norm(H_est[1] - H_anal[1])
            );
            fails += 1;
        }
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_ghf_jacobian();
    fails += test_ghf_hessian();

    if (fails > 0) {
        printf("[estimator/ghf.cpp] %d tests failed!\n", fails);
    } else {
        printf("[estimator/ghf.cpp] All tests passed!\n");
    }

    return fails;
}
