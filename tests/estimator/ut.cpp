#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <chrono>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "cugsplat/estimator/ut.h"

using namespace cugsplat::ut;

struct AuxData {};

// Test function that performs a simple linear transformation
template <int N, int M> struct LinearTransform {
    glm::mat<N, M, float>
        A; // N columns, M rows for N->M transformation (GLM is column-major)
    glm::vec<M, float> b;

    LinearTransform(glm::mat<N, M, float> const &A, glm::vec<M, float> const &b)
        : A(A), b(b) {}

    GSPLAT_HOST_DEVICE auto operator()(glm::vec<N, float> const &x
    ) const -> std::tuple<glm::vec<M, float>, bool, AuxData> {
        return {A * x + b, true, AuxData{}};
    }
};

// Test function that performs a quadratic transformation
template <int N, int M> struct QuadraticTransform {
    glm::mat<N, M, float>
        A; // N columns, M rows for N->M transformation (GLM is column-major)
    glm::vec<M, float> b;
    float scale;

    QuadraticTransform(
        glm::mat<N, M, float> const &A, glm::vec<M, float> const &b, float scale
    )
        : A(A), b(b), scale(scale) {}

    GSPLAT_HOST_DEVICE auto operator()(glm::vec<N, float> const &x
    ) const -> std::tuple<glm::vec<M, float>, bool, AuxData> {
        auto y = A * x + b;
        for (int i = 0; i < M; i++) {
            y[i] += scale * x[i] * x[i];
        }
        return {y, true, AuxData{}};
    }
};

// Test function that sometimes fails
template <int N, int M> struct FailingTransform {
    glm::mat<N, M, float>
        A; // N columns, M rows for N->M transformation (GLM is column-major)
    glm::vec<M, float> b;
    float threshold;

    FailingTransform(
        glm::mat<N, M, float> const &A, glm::vec<M, float> const &b, float threshold
    )
        : A(A), b(b), threshold(threshold) {}

    GSPLAT_HOST_DEVICE auto operator()(glm::vec<N, float> const &x
    ) const -> std::tuple<glm::vec<M, float>, bool, AuxData> {
        // Fail if any component of x is above threshold
        for (int i = 0; i < N; i++) {
            if (std::abs(x[i]) > threshold) {
                return {glm::vec<M, float>{}, false, AuxData{}};
            }
        }
        return {A * x + b, true, AuxData{}};
    }
};

int test_linear_transform() {
    int failures = 0;

    // Test case 1: Simple 2D to 2D linear transform
    {
        auto A = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);
        auto b = glm::vec2(1.0f, 2.0f);
        auto f = LinearTransform<2, 2>(A, b);

        auto mu = glm::vec2(0.0f, 0.0f);
        auto sqrt_covar = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);

        auto [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        if (!success) {
            printf("Linear transform test 1 failed: transform returned false\n");
            failures++;
        }

        // For linear transform, UT should give exact results
        auto expected_mu = A * mu + b;
        if (!is_close(mu_ut, expected_mu)) {
            printf("Linear transform test 1 failed: mean mismatch\n");
            printf("Expected: %s\n", glm::to_string(expected_mu).c_str());
            printf("Got: %s\n", glm::to_string(mu_ut).c_str());
            failures++;
        }

        auto expected_covar = A * glm::mat2x2(1.0f) * glm::transpose(A);
        if (!is_close(covar_ut, expected_covar)) {
            printf("Linear transform test 1 failed: covariance mismatch\n");
            printf("Expected: %s\n", glm::to_string(expected_covar).c_str());
            printf("Got: %s\n", glm::to_string(covar_ut).c_str());
            failures++;
        }
    }

    // Test case 2: 3D to 2D linear transform
    {
        auto A = glm::mat3x2(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
        auto b = glm::vec2(1.0f, 2.0f);
        auto f = LinearTransform<3, 2>(A, b);

        auto mu = glm::vec3(0.0f, 0.0f, 0.0f);
        auto sqrt_covar = glm::mat3x3(1.0f);

        auto [mu_ut, covar_ut, success, aux] =
            transform<3, 2, AuxData>(f, mu, sqrt_covar);

        if (!success) {
            printf("Linear transform test 2 failed: transform returned false\n");
            failures++;
        }

        // For 3D to 2D case, we need to handle dimensions correctly
        auto expected_mu = glm::vec2(
            A[0][0] * mu[0] + A[1][0] * mu[1] + A[2][0] * mu[2] + b[0],
            A[0][1] * mu[0] + A[1][1] * mu[1] + A[2][1] * mu[2] + b[1]
        );
        if (!is_close(mu_ut, expected_mu)) {
            printf("Linear transform test 2 failed: mean mismatch\n");
            printf("Expected: %s\n", glm::to_string(expected_mu).c_str());
            printf("Got: %s\n", glm::to_string(mu_ut).c_str());
            failures++;
        }

        // For covariance, we need to handle the 3D to 2D transformation
        // For y = Ax + b, cov(y) = A * cov(x) * A^T
        // Since cov(x) is identity, we just need A * A^T
        auto expected_covar = glm::mat2x2(0.0f);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 3; k++) {
                    expected_covar[i][j] +=
                        A[k][i] * A[k][j]; // Note: A is column-major
                }
            }
        }
        if (!is_close(covar_ut, expected_covar)) {
            printf("Linear transform test 2 failed: covariance mismatch\n");
            printf("Expected: %s\n", glm::to_string(expected_covar).c_str());
            printf("Got: %s\n", glm::to_string(covar_ut).c_str());
            failures++;
        }
    }

    return failures;
}

int test_quadratic_transform() {
    int failures = 0;

    // Test case 1: Simple 2D to 2D quadratic transform
    {
        auto A = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);
        auto b = glm::vec2(1.0f, 2.0f);
        auto f = QuadraticTransform<2, 2>(A, b, 0.1f);

        auto mu = glm::vec2(0.0f, 0.0f);
        auto sqrt_covar = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);

        auto [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        if (!success) {
            printf("Quadratic transform test 1 failed: transform returned false\n");
            failures++;
        }

        // For quadratic transform, UT should give approximate results
        // We can't check exact values, but we can verify the structure
        if (glm::length(mu_ut) < 1.0f) {
            printf("Quadratic transform test 1 failed: mean too small\n");
            printf("Got: %s\n", glm::to_string(mu_ut).c_str());
            failures++;
        }

        if (glm::determinant(covar_ut) < 0.0f) {
            printf("Quadratic transform test 1 failed: invalid covariance matrix\n");
            printf("Got: %s\n", glm::to_string(covar_ut).c_str());
            failures++;
        }
    }

    return failures;
}

int test_failing_transform() {
    int failures = 0;

    // Test case 1: Transform that fails for large inputs
    {
        auto A = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);
        auto b = glm::vec2(1.0f, 2.0f);
        auto f = FailingTransform<2, 2>(A, b, 2.0f);

        auto mu = glm::vec2(0.0f, 0.0f);
        auto sqrt_covar = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);

        auto [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        if (!success) {
            printf("Failing transform test 1 failed: transform should succeed for "
                   "small inputs\n");
            failures++;
        }
    }

    // Test case 2: Transform that fails for large inputs
    {
        auto A = glm::mat2x2(1.0f, 0.0f, 0.0f, 1.0f);
        auto b = glm::vec2(1.0f, 2.0f);
        auto f = FailingTransform<2, 2>(A, b, 0.5f);

        auto mu = glm::vec2(0.0f, 3.0f);
        auto sqrt_covar = glm::mat2x2(2.0f, 0.0f, 0.0f, 2.0f); // Larger covariance

        auto [mu_ut, covar_ut, success, aux] =
            transform<2, 2, AuxData>(f, mu, sqrt_covar);

        if (success) {
            printf("Failing transform test 2 failed: transform should fail for large "
                   "inputs\n");
            failures++;
        }
    }

    return failures;
}

int main() {
    int failures = 0;

    failures += test_linear_transform();
    failures += test_quadratic_transform();
    failures += test_failing_transform();

    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d tests failed!\n", failures);
    }

    return failures;
}