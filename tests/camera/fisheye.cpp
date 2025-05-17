#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <chrono>
#include <glm/gtx/string_cast.hpp>
#include <stdio.h>

#include "../helpers.h"
#include "curend/camera/fisheye.h"
#include "curend/estimator/ghq.h"

using namespace curend::fisheye;

// Test distortion and distortion_jac functions
auto test_distortion() -> int {
    int fails = 0;

    // Test case 1: No distortion (all coefficients zero)
    {
        auto const theta = 1.0f;
        auto const radial_coeffs = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
        auto const distorted = distortion(theta, radial_coeffs);
        auto const jac = distortion_jac(theta, radial_coeffs);
        auto const expected_distorted = theta;
        auto const expected_jac = 1.0f;

        if (!is_close(distorted, expected_distorted) || !is_close(jac, expected_jac)) {
            printf("\n=== Testing distortion ===\n");
            printf("\n[FAIL] Test 1: No distortion\n");
            printf("  Distorted: %f\n", distorted);
            printf("  Expected distorted: %f\n", expected_distorted);
            printf("  Jacobian: %f\n", jac);
            printf("  Expected Jacobian: %f\n", expected_jac);
            fails += 1;
        }
    }

    // Test case 2: Simple distortion
    {
        auto const theta = 1.0f;
        auto const radial_coeffs = std::array<float, 4>{0.1f, 0.0f, 0.0f, 0.0f};
        auto const distorted = distortion(theta, radial_coeffs);
        auto const jac = distortion_jac(theta, radial_coeffs);
        auto const expected_distorted = theta * (1.0f + 0.1f * theta * theta);
        auto const expected_jac = 1.0f + 3.0f * 0.1f * theta * theta;

        if (!is_close(distorted, expected_distorted) || !is_close(jac, expected_jac)) {
            printf("\n[FAIL] Test 2: Simple distortion\n");
            printf("  Distorted: %f\n", distorted);
            printf("  Expected distorted: %f\n", expected_distorted);
            printf("  Jacobian: %f\n", jac);
            printf("  Expected Jacobian: %f\n", expected_jac);
            fails += 1;
        }
    }

    return fails;
}

// Test undistortion function
auto test_undistortion() -> int {
    int fails = 0;

    // Test case 1: No distortion (all coefficients zero)
    {
        auto const theta_d = 1.0f;
        auto const radial_coeffs = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
        auto const [theta, valid] = undistortion(theta_d, radial_coeffs);
        auto const expected_theta = theta_d;

        if (!valid || !is_close(theta, expected_theta)) {
            printf("\n=== Testing undistortion ===\n");
            printf("\n[FAIL] Test 1: No distortion\n");
            printf("  Theta: %f\n", theta);
            printf("  Expected theta: %f\n", expected_theta);
            printf("  Valid: %d\n", valid);
            fails += 1;
        }
    }

    // Test case 2: Simple distortion
    {
        auto const theta_d = 1.1f;
        auto const radial_coeffs = std::array<float, 4>{0.1f, 0.0f, 0.0f, 0.0f};
        auto const [theta, valid] = undistortion(theta_d, radial_coeffs);
        auto const expected_theta = 1.0f;

        if (!valid || !is_close(theta, expected_theta, 1e-4f)) {
            printf("\n[FAIL] Test 2: Simple distortion\n");
            printf("  Theta: %f\n", theta);
            printf("  Expected theta: %f\n", expected_theta);
            printf("  Valid: %d\n", valid);
            fails += 1;
        }
    }

    return fails;
}

// Test monotonic_max_theta function
auto test_monotonic_max_theta() -> int {
    int fails = 0;

    // Test case 1: No distortion (all coefficients zero)
    {
        auto const radial_coeffs = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
        auto const max_theta = monotonic_max_theta(radial_coeffs);
        auto const expected_max_theta = std::numeric_limits<float>::max();

        if (!is_close(max_theta, expected_max_theta)) {
            printf("\n=== Testing monotonic_max_theta ===\n");
            printf("\n[FAIL] Test 1: No distortion\n");
            printf("  Max theta: %f\n", max_theta);
            printf("  Expected max theta: %f\n", expected_max_theta);
            fails += 1;
        }
    }

    // Test case 2: Simple distortion
    {
        auto const radial_coeffs = std::array<float, 4>{-0.1f, 0.0f, 0.0f, 0.0f};
        auto const max_theta = monotonic_max_theta(radial_coeffs);
        // For f'(theta) = 1 + 3*k1*theta^2 = 0
        // theta = sqrt(-1/(3*k1))
        auto const expected_max_theta = std::sqrt(-1.0f / (3.0f * -0.1f));

        if (!is_close(max_theta, expected_max_theta, 1e-4f)) {
            printf("\n[FAIL] Test 2: Simple distortion\n");
            printf("  Max theta: %f\n", max_theta);
            printf("  Expected max theta: %f\n", expected_max_theta);
            fails += 1;
        }
    }

    return fails;
}

// Test project function (perfect fisheye)
auto test_project() -> int {
    int fails = 0;

    // Test case 1: Point at image center
    {
        auto const camera_point = glm::fvec3(0.0f, 0.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const image_point = project(camera_point, focal_length, principal_point);
        auto const expected = principal_point;

        if (!is_close(image_point, expected)) {
            printf("\n=== Testing project (perfect fisheye) ===\n");
            printf("\n[FAIL] Test 1: Point at image center\n");
            printf("  Image point: %s\n", glm::to_string(image_point).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Point at 45 degrees
    {
        auto const camera_point = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const image_point = project(camera_point, focal_length, principal_point);
        auto const r = std::sqrt(2.0f);
        auto const theta = std::atan(r);
        auto const expected =
            principal_point + focal_length * (theta / r) * glm::fvec2(1.0f, 1.0f);

        if (!is_close(image_point, expected, 1e-4f)) {
            printf("\n[FAIL] Test 2: Point at 45 degrees\n");
            printf("  Image point: %s\n", glm::to_string(image_point).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

// Test project function (distorted fisheye)
auto test_project_distorted() -> int {
    int fails = 0;

    // Test case 1: Point at image center
    {
        auto const camera_point = glm::fvec3(0.0f, 0.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const radial_coeffs = std::array<float, 4>{0.1f, 0.0f, 0.0f, 0.0f};
        auto const [image_point, valid] =
            project(camera_point, focal_length, principal_point, radial_coeffs);
        auto const expected = principal_point;

        if (!valid || !is_close(image_point, expected)) {
            printf("\n=== Testing project (distorted fisheye) ===\n");
            printf("\n[FAIL] Test 1: Point at image center\n");
            printf("  Image point: %s\n", glm::to_string(image_point).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            printf("  Valid: %d\n", valid);
            fails += 1;
        }
    }

    // Test case 2: Point at 45 degrees
    {
        auto const camera_point = glm::fvec3(1.0f, 1.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const radial_coeffs = std::array<float, 4>{0.1f, 0.0f, 0.0f, 0.0f};
        auto const [image_point, valid] =
            project(camera_point, focal_length, principal_point, radial_coeffs);
        auto const r = std::sqrt(2.0f);
        auto const theta = std::atan(r);
        auto const theta_d = distortion(theta, radial_coeffs);
        auto const expected =
            principal_point + focal_length * (theta_d / r) * glm::fvec2(1.0f, 1.0f);

        if (!valid || !is_close(image_point, expected, 1e-4f)) {
            printf("\n[FAIL] Test 2: Point at 45 degrees\n");
            printf("  Image point: %s\n", glm::to_string(image_point).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            printf("  Valid: %d\n", valid);
            fails += 1;
        }
    }

    return fails;
}

// Test unproject function (perfect fisheye)
auto test_unproject() -> int {
    int fails = 0;

    // Test case 1: Point at image center
    {
        auto const image_point = glm::fvec2(320.0f, 240.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const dir = unproject(image_point, focal_length, principal_point);
        auto const expected = glm::fvec3(0.0f, 0.0f, 1.0f);

        if (!is_close(dir, expected)) {
            printf("\n=== Testing unproject (perfect fisheye) ===\n");
            printf("\n[FAIL] Test 1: Point at image center\n");
            printf("  Direction: %s\n", glm::to_string(dir).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    // Test case 2: Point at 45 degrees
    {
        auto const r = std::sqrt(2.0f);
        auto const theta = std::atan(r);
        auto const image_point =
            glm::fvec2(320.0f, 240.0f) +
            glm::fvec2(100.0f, 100.0f) * (theta / r) * glm::fvec2(1.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const dir = unproject(image_point, focal_length, principal_point);
        auto const expected = glm::normalize(glm::fvec3(1.0f, 1.0f, 1.0f));

        if (!is_close(dir, expected, 1e-4f)) {
            printf("\n[FAIL] Test 2: Point at 45 degrees\n");
            printf("  Direction: %s\n", glm::to_string(dir).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

// Test unproject function (distorted fisheye)
auto test_unproject_distorted() -> int {
    int fails = 0;

    // Test case 1: Point at image center
    {
        auto const image_point = glm::fvec2(320.0f, 240.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const radial_coeffs = std::array<float, 4>{0.1f, 0.0f, 0.0f, 0.0f};
        auto const [dir, valid] =
            unproject(image_point, focal_length, principal_point, radial_coeffs);
        auto const expected = glm::fvec3(0.0f, 0.0f, 1.0f);

        if (!valid || !is_close(dir, expected)) {
            printf("\n=== Testing unproject (distorted fisheye) ===\n");
            printf("\n[FAIL] Test 1: Point at image center\n");
            printf("  Direction: %s\n", glm::to_string(dir).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            printf("  Valid: %d\n", valid);
            fails += 1;
        }
    }

    // Test case 2: Point at 45 degrees
    {
        auto const r = std::sqrt(2.0f);
        auto const theta = std::atan(r);
        auto const radial_coeffs = std::array<float, 4>{0.1f, 0.0f, 0.0f, 0.0f};
        auto const theta_d = distortion(theta, radial_coeffs);
        auto const image_point =
            glm::fvec2(320.0f, 240.0f) +
            glm::fvec2(100.0f, 100.0f) * (theta_d / r) * glm::fvec2(1.0f, 1.0f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);
        auto const [dir, valid] =
            unproject(image_point, focal_length, principal_point, radial_coeffs);
        auto const expected = glm::normalize(glm::fvec3(1.0f, 1.0f, 1.0f));

        if (!valid || !is_close(dir, expected, 1e-4f)) {
            printf("\n[FAIL] Test 2: Point at 45 degrees\n");
            printf("  Direction: %s\n", glm::to_string(dir).c_str());
            printf("  Expected: %s\n", glm::to_string(expected).c_str());
            printf("  Valid: %d\n", valid);
            fails += 1;
        }
    }

    return fails;
}

// Test project_jac function (perfect fisheye)
auto test_project_jac() -> int {
    int fails = 0;

    {
        auto const camera_point = glm::fvec3(2.0f, 1.2f, 1.1f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);

        // Compute analytical gradient
        auto const J = project_jac(camera_point, focal_length);
        auto const v_image_point = glm::fvec2(0.5f, 0.8f);
        auto const v_camera_point = glm::transpose(J) * v_image_point;

        // Compute numerical gradient
        auto const expected =
            numerical_gradient(camera_point, [&](const glm::fvec3 &camera_point) {
                auto const image_point =
                    project(camera_point, focal_length, principal_point);
                return glm::dot(v_image_point, image_point);
            });

        if (!is_close(v_camera_point, expected)) {
            printf("\n[FAIL] Test 1: Gradient\n");
            printf("  Analytical: %s\n", glm::to_string(v_camera_point).c_str());
            printf("  Numerical: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

// Test project_hess function (perfect fisheye)
auto test_project_hess() -> int {
    int fails = 0;

    {
        auto const camera_point = glm::fvec3(2.0f, 1.2f, 1.1f);
        auto const focal_length = glm::fvec2(100.0f, 100.0f);
        auto const principal_point = glm::fvec2(320.0f, 240.0f);

        // Compute analytical gradient
        auto const &[H1, H2] = project_hess(camera_point, focal_length);
        auto const v_J1 = glm::fvec3(0.5f, 0.8f, 0.2f); // d((x, y, z) -> u)
        auto const v_J2 = glm::fvec3(0.3f, 0.4f, 0.6f); // d((x, y, z) -> v)
        auto const v_camera_point = H1 * v_J1 + H2 * v_J2;

        // Compute numerical gradient
        auto const expected = numerical_gradient(
            camera_point,
            [&](const glm::fvec3 &camera_point) {
                auto const J = project_jac(camera_point, focal_length);
                auto const J1 = glm::fvec3(J[0][0], J[1][0], J[2][0]);
                auto const J2 = glm::fvec3(J[0][1], J[1][1], J[2][1]);
                return glm::dot(v_J1, J1) + glm::dot(v_J2, J2);
            },
            1e-4f
        );

        // // Use GHQ to estimate gradient
        // auto const &[J_est, H_est] = curend::ghq::estimate_jacobian_and_hessian<3,
        // 2>(
        //     [&](const glm::fvec3 &camera_point) {
        //         return project(camera_point, focal_length, principal_point);
        //     },
        //     camera_point,
        //     glm::fvec3(1e-2f, 1e-2f, 1e-1f)
        // );
        // printf("H_est 0: %s\n", glm::to_string(std::get<0>(H_est)).c_str());
        // printf("H1: %s\n", glm::to_string(H1).c_str());
        // printf("H_est 1: %s\n", glm::to_string(std::get<1>(H_est)).c_str());
        // printf("H2: %s\n", glm::to_string(H2).c_str());

        // // Compare time cost between GHQ and analytical gradient
        // {
        //     auto const start_time = std::chrono::high_resolution_clock::now();
        //     for (int i = 0; i < 10000; ++i) {
        //         auto const &[J_est, H_est] =
        //         curend::ghq::estimate_jacobian_and_hessian<3, 2>(
        //             [&](const glm::fvec3 &camera_point) {
        //                 return project(camera_point, focal_length, principal_point);
        //             },
        //             camera_point, glm::fvec3(1e-2f, 1e-2f, 1e-1f));
        //     }
        //     auto const end_time = std::chrono::high_resolution_clock::now();
        //     auto const duration =
        //     std::chrono::duration_cast<std::chrono::microseconds>(end_time -
        //     start_time).count(); printf("GHQ time cost: %f ms\n", duration /
        //     10000.0f);

        //     auto const start_time_numerical =
        //     std::chrono::high_resolution_clock::now();
        // }
        // {
        //     auto const start_time = std::chrono::high_resolution_clock::now();
        //     for (int i = 0; i < 10000; ++i) {
        //         auto const &[H1, H2] = project_hess(camera_point, focal_length);
        //     }
        //     auto const end_time = std::chrono::high_resolution_clock::now();
        //     auto const duration =
        //     std::chrono::duration_cast<std::chrono::microseconds>(end_time -
        //     start_time).count(); printf("Analytical time cost: %f ms\n", duration /
        //     10000.0f);
        // }

        if (!is_close(v_camera_point, expected)) {
            printf("\n[FAIL] Test 1: Gradient\n");
            printf("  Analytical: %s\n", glm::to_string(v_camera_point).c_str());
            printf("  Numerical: %s\n", glm::to_string(expected).c_str());
            fails += 1;
        }
    }

    return fails;
}

auto main() -> int {
    int fails = 0;

    fails += test_distortion();
    fails += test_undistortion();
    fails += test_monotonic_max_theta();
    fails += test_project();
    fails += test_project_jac();
    fails += test_project_hess();
    fails += test_project_distorted();
    fails += test_unproject();
    fails += test_unproject_distorted();

    if (fails > 0) {
        printf("\nTotal number of failures: %d\n", fails);
        return 1;
    }

    printf("[camera/fisheye.cpp] All tests passed!\n");
    return 0;
}