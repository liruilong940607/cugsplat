#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <cmath>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "gaussian/utils.h"

using namespace gsplat;

void test_quat_to_rotmat_vjp() {
    printf("\n=== Testing quat_to_rotmat_vjp ===\n");

    // Test case 1: Identity quaternion
    {
        auto const quat = glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
        auto const v_R = glm::fmat3(1.0f);

        // Compute analytical gradient
        auto const v_quat = quat_to_rotmat_vjp(quat, v_R);

        // Compute numerical gradient
        auto const v_quat_num =
            numerical_gradient(quat, [&](const glm::fvec4 &q) {
                auto const R = quat_to_rotmat(q);
                return glm::dot(glm::fvec3(v_R[0]), R[0]) +
                       glm::dot(glm::fvec3(v_R[1]), R[1]) +
                       glm::dot(glm::fvec3(v_R[2]), R[2]);
            });

        if (!is_close(v_quat, v_quat_num)) {
            printf("\n[FAIL] Test 1: Identity quaternion\n");
            printf(
                "  Analytical gradient: %s\n", glm::to_string(v_quat).c_str()
            );
            printf(
                "  Numerical gradient: %s\n", glm::to_string(v_quat_num).c_str()
            );
            printf("  Error: %f\n", glm::length(v_quat - v_quat_num));
        }
    }

    // Test case 2: Random quaternion
    {
        auto const quat = glm::normalize(glm::fvec4(0.5f, 0.3f, 0.2f, 0.1f));
        auto const v_R = glm::fmat3(1.0f);

        // Compute analytical gradient
        auto const v_quat = quat_to_rotmat_vjp(quat, v_R);

        // Compute numerical gradient
        auto const v_quat_num =
            numerical_gradient(quat, [&](const glm::fvec4 &q) {
                auto const R = quat_to_rotmat(q);
                return glm::dot(glm::fvec3(v_R[0]), R[0]) +
                       glm::dot(glm::fvec3(v_R[1]), R[1]) +
                       glm::dot(glm::fvec3(v_R[2]), R[2]);
            });

        if (!is_close(v_quat, v_quat_num)) {
            printf("\n[FAIL] Test 2: Random quaternion\n");
            printf(
                "  Analytical gradient: %s\n", glm::to_string(v_quat).c_str()
            );
            printf(
                "  Numerical gradient: %s\n", glm::to_string(v_quat_num).c_str()
            );
            printf("  Error: %f\n", glm::length(v_quat - v_quat_num));
        }
    }
}

void test_quat_scale_to_scaled_rotmat_vjp() {
    printf("\n=== Testing quat_scale_to_scaled_rotmat_vjp ===\n");

    // Test case 1: Identity quaternion and scale
    {
        auto const quat = glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
        auto const scale = glm::fvec3(1.0f);
        auto const v_M = glm::fmat3(1.0f);

        // Compute analytical gradient
        auto const [v_quat, v_scale] =
            quat_scale_to_scaled_rotmat_vjp(quat, scale, v_M);

        // Compute numerical gradient for quaternion
        auto const v_quat_num =
            numerical_gradient(quat, [&](const glm::fvec4 &q) {
                auto const M = quat_scale_to_scaled_rotmat(q, scale);
                return glm::dot(glm::fvec3(v_M[0]), M[0]) +
                       glm::dot(glm::fvec3(v_M[1]), M[1]) +
                       glm::dot(glm::fvec3(v_M[2]), M[2]);
            });

        // Compute numerical gradient for scale
        auto const v_scale_num =
            numerical_gradient(scale, [&](const glm::fvec3 &s) {
                auto const M = quat_scale_to_scaled_rotmat(quat, s);
                return glm::dot(glm::fvec3(v_M[0]), M[0]) +
                       glm::dot(glm::fvec3(v_M[1]), M[1]) +
                       glm::dot(glm::fvec3(v_M[2]), M[2]);
            });

        if (!is_close(v_quat, v_quat_num) || !is_close(v_scale, v_scale_num)) {
            printf("\n[FAIL] Test 1: Identity quaternion and scale\n");
            printf(
                "  Analytical quat gradient: %s\n",
                glm::to_string(v_quat).c_str()
            );
            printf(
                "  Numerical quat gradient: %s\n",
                glm::to_string(v_quat_num).c_str()
            );
            printf("  Quat error: %f\n", glm::length(v_quat - v_quat_num));
            printf(
                "  Analytical scale gradient: %s\n",
                glm::to_string(v_scale).c_str()
            );
            printf(
                "  Numerical scale gradient: %s\n",
                glm::to_string(v_scale_num).c_str()
            );
            printf("  Scale error: %f\n", glm::length(v_scale - v_scale_num));
        }
    }

    // Test case 2: Random quaternion and scale
    {
        auto const quat = glm::normalize(glm::fvec4(0.5f, 0.3f, 0.2f, 0.1f));
        auto const scale = glm::fvec3(2.0f, 3.0f, 4.0f);
        auto const v_M = glm::fmat3(1.0f);

        // Compute analytical gradient
        auto const [v_quat, v_scale] =
            quat_scale_to_scaled_rotmat_vjp(quat, scale, v_M);

        // Compute numerical gradient for quaternion
        auto const v_quat_num =
            numerical_gradient(quat, [&](const glm::fvec4 &q) {
                auto const M = quat_scale_to_scaled_rotmat(q, scale);
                return glm::dot(glm::fvec3(v_M[0]), M[0]) +
                       glm::dot(glm::fvec3(v_M[1]), M[1]) +
                       glm::dot(glm::fvec3(v_M[2]), M[2]);
            });

        // Compute numerical gradient for scale
        auto const v_scale_num =
            numerical_gradient(scale, [&](const glm::fvec3 &s) {
                auto const M = quat_scale_to_scaled_rotmat(quat, s);
                return glm::dot(glm::fvec3(v_M[0]), M[0]) +
                       glm::dot(glm::fvec3(v_M[1]), M[1]) +
                       glm::dot(glm::fvec3(v_M[2]), M[2]);
            });

        if (!is_close(v_quat, v_quat_num) || !is_close(v_scale, v_scale_num)) {
            printf("\n[FAIL] Test 2: Random quaternion and scale\n");
            printf(
                "  Analytical quat gradient: %s\n",
                glm::to_string(v_quat).c_str()
            );
            printf(
                "  Numerical quat gradient: %s\n",
                glm::to_string(v_quat_num).c_str()
            );
            printf("  Quat error: %f\n", glm::length(v_quat - v_quat_num));
            printf(
                "  Analytical scale gradient: %s\n",
                glm::to_string(v_scale).c_str()
            );
            printf(
                "  Numerical scale gradient: %s\n",
                glm::to_string(v_scale_num).c_str()
            );
            printf("  Scale error: %f\n", glm::length(v_scale - v_scale_num));
        }
    }
}

void test_quat_scale_to_covar_vjp() {
    printf("\n=== Testing quat_scale_to_covar_vjp ===\n");

    // Test case 1: Identity quaternion and scale
    {
        auto const quat = glm::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
        auto const scale = glm::fvec3(1.0f);
        auto const v_covar = glm::fmat3(1.0f);

        // Compute analytical gradient
        auto const [v_quat, v_scale] =
            quat_scale_to_covar_vjp(quat, scale, v_covar);

        // Compute numerical gradient for quaternion
        auto const v_quat_num =
            numerical_gradient(quat, [&](const glm::fvec4 &q) {
                auto const covar = quat_scale_to_covar(q, scale);
                return glm::dot(glm::fvec3(v_covar[0]), covar[0]) +
                       glm::dot(glm::fvec3(v_covar[1]), covar[1]) +
                       glm::dot(glm::fvec3(v_covar[2]), covar[2]);
            });

        // Compute numerical gradient for scale
        auto const v_scale_num =
            numerical_gradient(scale, [&](const glm::fvec3 &s) {
                auto const covar = quat_scale_to_covar(quat, s);
                return glm::dot(glm::fvec3(v_covar[0]), covar[0]) +
                       glm::dot(glm::fvec3(v_covar[1]), covar[1]) +
                       glm::dot(glm::fvec3(v_covar[2]), covar[2]);
            });

        if (!is_close(v_quat, v_quat_num) || !is_close(v_scale, v_scale_num)) {
            printf("\n[FAIL] Test 1: Identity quaternion and scale\n");
            printf(
                "  Analytical quat gradient: %s\n",
                glm::to_string(v_quat).c_str()
            );
            printf(
                "  Numerical quat gradient: %s\n",
                glm::to_string(v_quat_num).c_str()
            );
            printf("  Quat error: %f\n", glm::length(v_quat - v_quat_num));
            printf(
                "  Analytical scale gradient: %s\n",
                glm::to_string(v_scale).c_str()
            );
            printf(
                "  Numerical scale gradient: %s\n",
                glm::to_string(v_scale_num).c_str()
            );
            printf("  Scale error: %f\n", glm::length(v_scale - v_scale_num));
        }
    }

    // Test case 2: Random quaternion and scale
    {
        auto const quat = glm::normalize(glm::fvec4(0.5f, 0.3f, 0.2f, 0.1f));
        auto const scale = glm::fvec3(0.2f, 0.3f, 0.4f);
        auto const v_covar = glm::fmat3(1.0f);

        // Compute analytical gradient
        auto const [v_quat, v_scale] =
            quat_scale_to_covar_vjp(quat, scale, v_covar);

        // Compute numerical gradient for quaternion
        auto const v_quat_num =
            numerical_gradient(quat, [&](const glm::fvec4 &q) {
                auto const covar = quat_scale_to_covar(q, scale);
                return glm::dot(glm::fvec3(v_covar[0]), covar[0]) +
                       glm::dot(glm::fvec3(v_covar[1]), covar[1]) +
                       glm::dot(glm::fvec3(v_covar[2]), covar[2]);
            });

        // Compute numerical gradient for scale
        auto const v_scale_num =
            numerical_gradient(scale, [&](const glm::fvec3 &s) {
                auto const covar = quat_scale_to_covar(quat, s);
                return glm::dot(glm::fvec3(v_covar[0]), covar[0]) +
                       glm::dot(glm::fvec3(v_covar[1]), covar[1]) +
                       glm::dot(glm::fvec3(v_covar[2]), covar[2]);
            });

        if (!is_close(v_quat, v_quat_num) || !is_close(v_scale, v_scale_num)) {
            printf("\n[FAIL] Test 2: Random quaternion and scale\n");
            printf(
                "  Analytical quat gradient: %s\n",
                glm::to_string(v_quat).c_str()
            );
            printf(
                "  Numerical quat gradient: %s\n",
                glm::to_string(v_quat_num).c_str()
            );
            printf("  Quat error: %f\n", glm::length(v_quat - v_quat_num));
            printf(
                "  Analytical scale gradient: %s\n",
                glm::to_string(v_scale).c_str()
            );
            printf(
                "  Numerical scale gradient: %s\n",
                glm::to_string(v_scale_num).c_str()
            );
            printf("  Scale error: %f\n", glm::length(v_scale - v_scale_num));
        }
    }
}

int main() {
    test_quat_to_rotmat_vjp();
    test_quat_scale_to_scaled_rotmat_vjp();
    test_quat_scale_to_covar_vjp();
    return 0;
}